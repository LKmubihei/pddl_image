(define (problem picture_208)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
         red_pump green_regulator blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear green_regulator)
        (clear blue_battery)
        (part_at red_pump table)
        (part_at green_regulator regulator_placement)
        (part_at blue_battery battery_placement)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)