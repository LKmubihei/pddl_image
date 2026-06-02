(define (problem picture_338)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_battery green_pump green_regulator blue_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear green_pump)
        (clear green_regulator)
        (clear blue_battery)
        (part_at green_pump table)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at green_regulator table)
        (part_at blue_battery buffer_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
        )
    )
)