(define (problem picture_201)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery blue_battery red_pump green_pump green_regulator blue_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
        (clear red_pump)
        (clear green_pump)
        (clear green_regulator)
        (clear blue_regulator)
        (part_at red_pump table)
        (part_at green_battery table)
        (part_at green_pump table)
        (part_at blue_regulator table)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)

    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)